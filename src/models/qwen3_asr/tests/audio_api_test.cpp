#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_asr/transcription.hpp"
#include "tests/support/model_api_test_support.hpp"

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
           qwen3_asr::TranscriptionResult* result, std::string* error) {
    ++calls;
    wav_bytes = request.wav.size();
    pcm_samples = request.pcm16k.size();
    pcm.assign(request.pcm16k.begin(), request.pcm16k.end());
    context = request.context;
    language = request.language;
    maximum_tokens = request.max_new_tokens;
    if (fail) {
      result->invalid_audio = invalid_audio;
      if (error != nullptr)
        *error = "Qwen3-ASR audio is truncated";
      return false;
    }
    result->language = "English";
    result->text = "A native transcript.";
    result->audio_samples = 32000;
    result->audio_tokens = 25;
    result->generated_ids = {1, 2, 3};
    result->timings.audio_encoder_ms = 10.0;
    result->timings.text_decoder_ms = 20.0;
    if (request.on_text && !request.on_text("A native")) {
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
    return true;
  }

  bool fail{false};
  bool invalid_audio{false};
  int calls{0};
  std::size_t wav_bytes{0};
  std::size_t pcm_samples{0};
  std::vector<float> pcm;
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

// 0.1 s of PCM16 silence at 8 kHz: 1,600 samples once resampled to 16 kHz.
std::string Wav() {
  constexpr std::uint32_t kRate = 8000;
  constexpr std::uint32_t kSamples = 800;
  std::string wav;
  const auto put = [&wav](std::uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i)
      wav += static_cast<char>((value >> (8 * i)) & 0xFFU);
  };
  wav += "RIFF";
  put(36 + 2 * kSamples, 4);
  wav += "WAVEfmt ";
  put(16, 4);
  put(1, 2);
  put(1, 2);
  put(kRate, 4);
  put(2 * kRate, 4);
  put(2, 2);
  put(16, 2);
  wav += "data";
  put(2 * kSamples, 4);
  wav.append(2 * kSamples, '\0');
  return wav;
}

std::string Multipart(std::string response_format = "json",
                      std::string_view extra_field = {},
                      std::string_view stream = {},
                      std::string_view file = {}) {
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
  const std::string wav = Wav();
  append("file", file.empty() ? std::string_view(wav) : file, "sample.wav");
  append("model", "qwen3-asr");
  append("language", "en");
  append("prompt", "Names: Gufo.");
  append("response_format", response_format);
  append("max_tokens", "42");
  append("temperature", "0");
  if (!stream.empty())
    append("stream", stream);
  if (!extra_field.empty()) {
    append(extra_field, "logprobs");
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
  Check(runner.calls == 1 && runner.wav_bytes == 0U &&
            runner.pcm_samples == 1600U && runner.context == "Names: Gufo." &&
            runner.language == std::optional<std::string>("en") &&
            runner.maximum_tokens == 42U,
        "multipart fields reach the model contract");
  Check(HasHeader(response, "X-Gufo-ASR-Backend", "native-hip"),
        "native backend is exposed");
  Check(response.log_details.find("encoder_ms=") != std::string::npos &&
            response.log_details.find("decoder_ms=") != std::string::npos &&
            response.log_details.find("A native transcript.") ==
                std::string::npos &&
            response.log_details.find("Names: Gufo.") == std::string::npos,
        "transcription diagnostics report work without transcript or context");
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
  Check(Send(service, Multipart("json", "typo")).status == 200,
        "unknown multipart fields are ignored");
  Check(Send(service, Multipart("json", "include[]")).status == 400,
        "logprobs requests fail closed");
  Check(Send(service, Multipart(), "application/json").status == 400,
        "non-multipart requests fail closed");
  Check(runner.calls == 3, "invalid requests never invoke inference");
}

void TestFailures() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  // Serve decodes the upload, so undecodable audio never reaches a provider.
  const HttpResponse unreadable =
      Send(service, Multipart("json", {}, {}, "RIFFfake"));
  const HttpResponse unreadable_stream =
      Send(service, Multipart("json", {}, "true", "not audio"));
  Check(unreadable.status == 400 && unreadable_stream.status == 400 &&
            unreadable.body.find("\"code\":\"invalid_audio_input\"") !=
                std::string::npos &&
            unreadable_stream.body.find("audio must be WAV, FLAC, MP3") !=
                std::string::npos &&
            runner.calls == 0,
        "undecodable uploads are client errors before inference");
  runner.fail = true;
  runner.invalid_audio = true;
  const HttpResponse malformed = Send(service, Multipart());
  Check(malformed.status == 400 &&
            malformed.body.find("\"code\":\"invalid_audio_input\"") !=
                std::string::npos &&
            malformed.body.find("audio is truncated") != std::string::npos,
        "undecodable audio is a client error");
  const auto stream = Send(service, Multipart("json", {}, "true"));
  std::string events;
  stream.streaming_body([&](std::string_view part) {
    events += part;
    return true;
  });
  Check(
      stream.stream_log->error_code == "invalid_audio_input" &&
          events.find("\"code\":\"invalid_audio_input\"") != std::string::npos,
      "streaming reports undecodable audio as invalid input");
  runner.invalid_audio = false;
  const HttpResponse failed = Send(service, Multipart());
  Check(failed.status == 500 &&
            failed.body.find("\"code\":\"transcription_failed\"") !=
                std::string::npos,
        "other runner failures stay server errors");
}

void TestStreamingAndChunks() {
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  const auto response = Send(service, Multipart("json", {}, "true"));
  Check(response.status == 200 && response.streaming_body && runner.calls == 0,
        "streaming defers inference until the response is consumed");
  std::string events;
  response.streaming_body([&](std::string_view part) {
    events += part;
    return true;
  });
  Check(
      runner.calls == 1 &&
          events.find("\"delta\":\"A native\"") != std::string::npos &&
          events.find("\"delta\":\" transcript.\"") != std::string::npos &&
          events.find("\"type\":\"transcript.text.done\"") != std::string::npos,
      "OpenAI SSE emits incremental deltas and a complete transcript");
  const auto disconnected = Send(service, Multipart("json", {}, "true"));
  int writes = 0;
  disconnected.streaming_body([&](std::string_view) {
    ++writes;
    return false;
  });
  Check(writes == 1 && disconnected.stream_log->error_code == "cancelled",
        "disconnected stream cancels inference without a done event");
  Check(!Send(service, Multipart("json", {}, "false")).streaming_body,
        "stream=false retains the ordinary JSON contract");
  Check(Send(service, Multipart("json", {}, "yes")).status == 400 &&
            Send(service, Multipart("text", {}, "true")).status == 400,
        "invalid streaming configurations fail closed");

  constexpr std::size_t limit = 4 * qwen3_asr::kAudioSampleRate;
  const auto short_chunk =
      qwen3_asr::SplitAudio(std::vector<float>(1600), 1600);
  Check(short_chunk.size() == 1 && short_chunk.front().samples == 1600,
        "a short unsplit input does not require a half-second chunk budget");
  std::vector<float> waveform(limit * 2 + 137, 0.25F);
  std::fill(waveform.begin() + 3 * qwen3_asr::kAudioSampleRate,
            waveform.begin() + 3 * qwen3_asr::kAudioSampleRate + 1600, 0.0F);
  const auto chunks = qwen3_asr::SplitAudio(waveform, limit);
  std::size_t consumed = 0;
  for (const auto& chunk : chunks) {
    Check(
        chunk.offset == consumed && chunk.samples > 0 && chunk.samples <= limit,
        "long audio partitions exactly within the hard capacity");
    consumed += chunk.samples;
  }
  Check(consumed == waveform.size() &&
            chunks.front().samples == 3 * qwen3_asr::kAudioSampleRate,
        "chunking prefers a quiet boundary without dropping samples");
  for (const std::size_t budget : {13U, 104U, 725U}) {
    const auto samples = qwen3_asr::AudioSamplesForTokenBudget(budget);
    Check(qwen3_asr::AudioEmbeddingTokenCount(samples / 160) <= budget &&
              qwen3_asr::AudioEmbeddingTokenCount((samples + 1) / 160) > budget,
          "audio capacity includes convolutional tails");
  }
}

std::string ReadFixture(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "missing audio fixture: " + path);
  return {std::istreambuf_iterator<char>(input), {}};
}

// Serve decodes uploads for every provider; Qwen3-ASR must see the samples
// its own decoder and resampler made from the same file, bit for bit.
void TestServeDecodeMatchesQwen3() {
  std::string stereo = Wav();
  // 44.1 kHz stereo PCM16 noise, so that the mono mix and resampler run.
  std::string noise;
  std::uint32_t state = 12345;
  for (int i = 0; i < 2 * 4410; ++i) {
    state = state * 1664525U + 1013904223U;
    noise += static_cast<char>(state >> 24);
    noise += static_cast<char>(state >> 16);
  }
  const auto put = [](std::string& wav, std::size_t at, std::uint32_t value) {
    for (int i = 0; i < 4; ++i)
      wav[at + i] = static_cast<char>((value >> (8 * i)) & 0xFFU);
  };
  stereo.resize(44);
  stereo += noise;
  put(stereo, 4, static_cast<std::uint32_t>(36 + noise.size()));
  stereo[22] = 2;
  put(stereo, 24, 44100);
  put(stereo, 28, 44100 * 4);
  stereo[32] = 4;
  put(stereo, 40, static_cast<std::uint32_t>(noise.size()));
  const std::vector<std::pair<std::string, std::string>> inputs = {
      {"stereo 44.1 kHz WAV", stereo},
      {"MP3", ReadFixture(std::string(GUFO_QWEN3_ASR_SPEECH_FIXTURES) +
                          "/clear-english-voice.mp3")},
      {"Ogg Vorbis", ReadFixture(std::string(GUFO_AUDIO_DECODE_FIXTURES) +
                                 "/vorbis-first.ogg")},
  };
  FakeRunner runner;
  AsrService service = MakeService(&runner);
  for (const auto& [name, file] : inputs) {
    const auto bytes = std::as_bytes(std::span(file.data(), file.size()));
    qwen3_asr::AudioBuffer qwen;
    std::string error;
    Check(qwen3_asr::DecodeWav(bytes, &qwen, &error), name + ": " + error);
    Check(Send(service, Multipart("json", {}, {}, file)).status == 200,
          name + ": upload is transcribed");
    Check(runner.pcm == qwen3_asr::ResampleMono16k(qwen),
          name + ": serve and Qwen3-ASR decode the same samples");
  }
}

}  // namespace

int main() {
  TestServeDecodeMatchesQwen3();
  TestJsonRequest();
  TestFormatsAndValidation();
  TestFailures();
  TestStreamingAndChunks();
  std::cout << "PASS qwen3_asr_audio_api_test\n";
  return 0;
}
