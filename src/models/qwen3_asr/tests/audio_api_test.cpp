#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <utility>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"

namespace {

namespace qwen3_asr = gufo::models::qwen3_asr;
using gufo::server::AsrService;
using gufo::server::AsrServiceOptions;
using gufo::server::HandleAudioAsrApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_asr_audio_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

class FakeRunner {
public:
  bool Run(const qwen3_asr::TranscriptionRequest& request,
           const qwen3_asr::CancellationCheck&,
           qwen3_asr::TranscriptionResult* result, std::string*) {
    ++calls;
    wav_bytes = request.wav.size();
    context = request.context;
    language = request.language;
    maximum_tokens = request.max_new_tokens;
    result->language = "English";
    result->text = "A native transcript.";
    result->audio_samples = 32000;
    result->audio_tokens = 25;
    result->generated_ids = {1, 2, 3};
    result->timings.audio_encoder_ms = 10.0;
    result->timings.text_decoder_ms = 20.0;
    return true;
  }

  int calls{0};
  std::size_t wav_bytes{0};
  std::string context;
  std::optional<std::string> language;
  std::size_t maximum_tokens{0};
};

AsrService MakeService(FakeRunner* runner) {
  return AsrService(AsrServiceOptions{
      .model_root = {},
      .validate_model = false,
      .model_id = "qwen3-asr-1.7b",
      .runner =
          [runner](const qwen3_asr::TranscriptionRequest& request,
                   const qwen3_asr::CancellationCheck& cancelled,
                   qwen3_asr::TranscriptionResult* result, std::string* error) {
            return runner->Run(request, cancelled, result, error);
          },
  });
}

std::string Multipart(std::string response_format = "json",
                      bool include_unknown = false) {
  constexpr std::string_view boundary = "gufo-test-boundary";
  std::string body;
  const auto append = [&](std::string_view name, std::string_view value,
                          std::string_view filename = {}) {
    body += "--";
    body += boundary;
    body += "\r\nContent-Disposition: form-data; name=\"";
    body += name;
    body += '"';
    if (!filename.empty()) {
      body += "; filename=\"";
      body += filename;
      body += '"';
    }
    body += "\r\n\r\n";
    body.append(value.data(), value.size());
    body += "\r\n";
  };
  append("file", "RIFFfake", "sample.wav");
  append("model", "qwen3-asr");
  append("language", "en");
  append("prompt", "Names: Gufo.");
  append("response_format", response_format);
  append("max_tokens", "42");
  append("temperature", "0");
  if (include_unknown) {
    append("typo", "true");
  }
  body += "--";
  body += boundary;
  body += "--\r\n";
  return body;
}

HttpResponse Send(AsrService& service, std::string body,
                  std::string content_type =
                      "multipart/form-data; boundary=gufo-test-boundary") {
  return HandleAudioAsrApiRequest(
      HttpRequest{
          .method = "POST",
          .path = "/v1/audio/transcriptions",
          .query = {},
          .body = std::move(body),
          .headers = {{"Content-Type", std::move(content_type)}},
          .is_cancelled = {},
      },
      service);
}

bool HasHeader(const HttpResponse& response, std::string_view name,
               std::string_view value) {
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return header.first == name && header.second == value;
  });
}

void TestJsonRequest() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse response = Send(service, Multipart());
  Check(response.status == 200 &&
            response.body.find("\"text\":\"A native transcript.\"") !=
                std::string::npos,
        "valid multipart request returns OpenAI JSON");
  Check(runner.calls == 1 && runner.wav_bytes == 8U &&
            runner.context == "Names: Gufo." &&
            runner.language == std::optional<std::string>("en") &&
            runner.maximum_tokens == 42U,
        "multipart fields reach the model contract");
  Check(HasHeader(response, "X-Gufo-ASR-Backend", "native-hip"),
        "native backend is exposed");
}

void TestFormatsAndValidation() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const HttpResponse text = Send(service, Multipart("text"));
  Check(text.status == 200 && text.body == "A native transcript." &&
            HasHeader(text, "Content-Type", "text/plain; charset=utf-8"),
        "text response format is supported");
  const HttpResponse verbose = Send(service, Multipart("verbose_json"));
  Check(verbose.status == 200 &&
            verbose.body.find("\"duration\":2") != std::string::npos &&
            verbose.body.find("\"language\":\"english\"") != std::string::npos,
        "verbose_json exposes duration and language");
  Check(Send(service, Multipart("srt")).status == 400,
        "unsupported response formats fail closed");
  Check(Send(service, Multipart("json", true)).status == 400,
        "unknown multipart fields fail closed");
  Check(Send(service, Multipart(), "application/json").status == 400,
        "non-multipart requests fail closed");
  Check(runner.calls == 2, "invalid requests never invoke inference");
}

}  // namespace

int main() {
  TestJsonRequest();
  TestFormatsAndValidation();
  std::cout << "PASS qwen3_asr_audio_api_test\n";
  return 0;
}
